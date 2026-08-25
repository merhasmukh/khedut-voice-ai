"""
Automated test suite for Database and Context Management in Khedut Voice AI.
"""

import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_db, AsyncSessionLocal
from database import crud
from database.models import FarmerProfile, Conversation, Message


async def run_tests():
    print("🧪 [1/5] Initializing database tables...")
    await init_db()

    async with AsyncSessionLocal() as db:
        print("🧪 [2/5] Testing FarmerProfile CRUD...")
        test_profile = FarmerProfile()
        db.add(test_profile)
        await db.commit()
        await db.refresh(test_profile)
        assert test_profile is not None, "Profile should not be None"
        assert test_profile.id is not None, "Profile ID should be set"
        print(f"   ✓ Test profile created with ID: {test_profile.id}")

        # Update profile
        updated_profile = await crud.update_farmer_profile(
            db,
            test_profile.id,
            {"village": "ગોંડલ", "land_acres": 8.5, "crops": ["કપાસ", "મગફળી", "સોયાબીન", "તલ"]},
        )
        assert updated_profile.village == "ગોંડલ"
        assert updated_profile.land_acres == 8.5
        assert "સોયાબીન" in updated_profile.crops
        print(f"   ✓ Profile updated successfully: {updated_profile.village}, {updated_profile.land_acres} acres")

        print("🧪 [3/5] Testing Conversation and Message CRUD...")
        conv = await crud.get_or_create_conversation(db, title="કપાસમાં ગુલાબી ઈયળનું નિયંત્રણ")
        assert conv is not None
        assert conv.id is not None
        print(f"   ✓ Conversation created: {conv.id} ('{conv.title}')")

        # Update this conversation's specific profile
        await crud.update_farmer_profile(
            db,
            conv.farmer_id,
            {"village": "ગોંડલ", "land_acres": 8.5, "crops": ["કપાસ", "મગફળી", "સોયાબીન", "તલ"]},
        )

        # Add message turns
        msg1 = await crud.add_message(
            db,
            conversation_id=conv.id,
            role="user",
            content="મારા કપાસમાં ગુલાબી ઈયળ આવી છે, પ્રાકૃતિક ઉપાય શું કરવો?",
        )
        assert msg1.id is not None
        assert msg1.role == "user"

        msg2 = await crud.add_message(
            db,
            conversation_id=conv.id,
            role="assistant",
            content="ગુલાબી ઈયળ માટે લીંબોળીનું તેલ (Neem Oil 10,000 PPM) અથવા બ્રહ્માસ્ત્રનો છંટકાવ કરવો અને ફેરોમોન ટ્રેપ ગોઠવવા.",
            audio_format="pcm_24000",
        )
        assert msg2.id is not None
        assert msg2.role == "assistant"
        print("   ✓ User and Assistant turns added to conversation")

        # Query conversation with messages
        queried_conv = await crud.get_conversation(db, conv.id)
        assert queried_conv is not None
        assert len(queried_conv.messages) >= 2
        print(f"   ✓ Fetched conversation with {len(queried_conv.messages)} messages")

        print("🧪 [4/5] Testing Dynamic Prompt Context Builder...")
        context_str = await crud.build_conversation_context(db, conversation_id=conv.id)
        assert "ગોંડલ" in context_str, "Context should contain village name"
        assert "8.5" in context_str, "Context should contain land acres"
        assert "ગુલાબી ઈયળ" in context_str, "Context should contain previous conversation topic"
        print("   ✓ Context compiled successfully:")
        print("   -------------------------------------------------------")
        print("   " + "\n   ".join(context_str.split("\n")[:8]))
        print("   ...")
        print("   -------------------------------------------------------")

        print("🧪 [5/5] Testing Conversation Listing and Cleanup...")
        all_convs = await crud.list_conversations(db)
        assert len(all_convs) >= 1
        print(f"   ✓ Listed {len(all_convs)} conversations from database")

        # Clean up test conversation and test profile so database remains pristine
        await crud.delete_conversation(db, conv.id)
        await db.delete(test_profile)
        await db.commit()
        print("   ✓ Cleaned up test session and test profile from database")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(run_tests())
